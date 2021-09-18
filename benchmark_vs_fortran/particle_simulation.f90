module ParticleSimulation

    integer, parameter :: dp = kind(0.0d0)

contains 

elemental real(dp) function wrap(x,side)
    implicit none
    real(dp), intent(in) :: x, side
    wrap = dmod(x,side)
    if (wrap >= side/2) then
        wrap = wrap - side
    else if (wrap < -side/2) then
        wrap = wrap + side
    end if
end function wrap

real(dp) function norm(ndim,x)
    integer :: ndim
    real(dp) :: x(ndim)
    norm = 0.0_dp
    do i = 1, ndim
        norm = norm + x(i)**2
    end do
    norm  = sqrt(norm)
end function norm

subroutine force_pair(ndim,fx,x,y,cutoff,side)
    implicit none
    integer :: ndim
    real(dp) :: fx(ndim), d
    real(dp) :: x(ndim), y(ndim), cutoff, side, dv(ndim)
    dv = wrap(y-x,side)
    d = norm(ndim,dv)
    if (d > cutoff) then
        fx = 0.0_dp
    else
        fx = (d - cutoff)*(dv/d)
    end if
end subroutine force_pair

subroutine forces(n,ndim,f,x,cutoff,side)
    implicit none
    integer :: n, ndim, i, j
    real(dp) :: f(ndim,n), x(ndim,n)
    real(dp) :: fx(ndim)
    real(dp) :: cutoff, side
    f = 0.0_dp
    do i = 1, n-1
        do j = i+1, n
            call force_pair(ndim,fx,x(:,i),x(:,j),cutoff,side)
            f(:,i) = f(:,i) + fx
            f(:,j) = f(:,j) - fx
        end do
    end do
end subroutine forces

real(dp) function dp_rand()
    call random_number(dp_rand)
end function dp_rand

subroutine md(n,ndim,x0,v0,mass,dt,nsteps,isave,trajectory,cutoff,side)
    implicit none
    integer :: n, ndim, i, step, nsteps, isave, isaved
    real(dp) :: dt
    real(dp) :: x0(ndim,n), v0(ndim,n), mass(n)
    real(dp) :: x(ndim,n), v(ndim,n), f(ndim,n), a(ndim,n)
    real(dp) :: trajectory(ndim,n,nsteps/isave+1)
    real(dp) :: cutoff, side
    ! Save initial positions
    trajectory(:,:,1) = x0
    x = x0
    v = v0
    isaved = 1
    do step = 1, nsteps
        ! Compute forces
        call forces(n,ndim,f,x,cutoff,side)
        ! Update positions and velocities 
        do i = 1, n
           a(:,i) = f(:,i) / mass(i)
           x(:,i) = x(:,i) + v(:,i)*dt + a(:,i)*dt**2/2
           v(:,i) = v(:,i) + a(:,i)*dt
        end do
        ! Save if required
        if (mod(step,isave) == 0) then
            isaved = isaved + 1
            trajectory(:,:,isaved) = x
        end if
    end do
end subroutine md

end module ParticleSimulation

program main
    use ParticleSimulation
    implicit none
    integer, parameter :: n = 100, ndim = 2
    integer :: i, j, k, nsteps, isave
    real(dp) :: x0(ndim,n), v0(ndim,n), mass(n)
    real(dp) :: dt
    real(dp) :: cutoff, side
    real(dp), allocatable :: trajectory(:,:,:)
    ! Initialize positions and velocities
    do i = 1, n
        do j = 1, ndim
            x0(j,i) = -50.0_dp + 100_dp*dp_rand()
            v0(j,i) = -1.0_dp + 2.0_dp*dp_rand()
        end do
        mass(i) = 10.0_dp
    end do
    ! Parameters
    dt = 0.001_dp
    nsteps = 200000
    isave = 1000
    cutoff = 5.0_dp
    side = 100.0_dp
    allocate(trajectory(ndim,n,nsteps/isave + 1))
    ! Run simulation
    call md(n,ndim,x0,v0,mass,dt,nsteps,isave,trajectory,cutoff,side)
    open(10,file="traj_fortran.xyz")
    k = 0
    do i = 1, nsteps/isave + 1
        k = k + 1
        write(10,*) n
        write(10,*) " step = ", k
        do j = 1, n
            write(10,*) "He", wrap(trajectory(1,j,k),side), wrap(trajectory(2,j,k),side), 0.0_dp
        end do
    end do
    close(10)
end program main

